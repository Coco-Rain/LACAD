import cadquery as cq

result = (
cq.Workplane("XZ")
.polygon(6, 80)
.extrude(150)
.faces("<Y")
.shell(-6)
.faces("<Y[1]")
.wires()
.translate((0, -1, 0))
.toPending()
.offset2D(-1)
.extrude(150, combine=False)
.faces(">Z or >>Z[-2]")
.shell(-3)
)
cq.exporters.export(result, 'GT.stl')