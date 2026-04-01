import cadquery as cq

result = (
cq.Workplane()
.hLine(3)
.line((1.7), 3)
.hLineTo(0)
.mirrorY()
.extrude(-2.7)
.faces("<Z")
.edges("<Y")
.workplane()
.transformed(rotate=(60, 0, 0))
.split(keepBottom=True)
)
cq.exporters.export(result, 'GT.stl')