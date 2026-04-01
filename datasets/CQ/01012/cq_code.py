import cadquery as cq

result = (
cq.Workplane("front")
.box(66, 10, 70)
.faces("<Y")
.workplane(centerOption = 'CenterOfBoundBox')
.move(0, -30)
.rect(66, 4)
.extrude(86)
.faces("<Z[1]")
.workplane()
.move(0, 5)
.rect(50, 50, forConstruction = True)
.vertices()
.cboreHole(5, 7.5, 2)
)
cq.exporters.export(result, 'GT.stl')